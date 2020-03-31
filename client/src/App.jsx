import React, { Component } from 'react';

import { Route, Switch, withRouter } from 'react-router-dom';

import Auth from 'pages/Auth';
import Home from 'pages/Home';
import Layout from 'components/Layout';
import NewFont from 'pages/NewFont';
import Saved from 'pages/Saved';

class App extends Component {
    state = {
        user: null
    };

    loginHandler = (token, expTime) => {
        localStorage.setItem('token', token);
        // localStorage.setItem(
        //     'expDate',
        //     new Date(new Date().getTime() + +expTime)
        // );
        this.setState({
            user: {
                token: token
            }
        });
        // setTimeout(() => {
        //     this.logoutHandler();
        // }, expTime);
    };

    logoutHandler = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('expDate');
        this.setState({
            user: null
        });
        this.props.history.push('/');
    };

    componentDidMount() {
        const token = localStorage.getItem('token');
        const expDate = new Date(localStorage.getItem('expDate'));
        if (token && expDate > new Date().getTime()) {
            this.loginHandler(token, expDate - new Date().getTime());
        }
    }

    render() {
        console.log(this.state);
        return (
            <Layout
                isAuth={Boolean(this.state.user)}
                logout={this.state.user ? this.logoutHandler : null}
            >
                <Switch>
                    <Route path="/" exact>
                        <Home />
                    </Route>
                    <Route path="/new-font" exact>
                        <NewFont user={this.state.user} />
                    </Route>
                    {!this.state.user ? (
                        <Route path="/auth">
                            <Auth login={this.loginHandler} />
                        </Route>
                    ) : null}
                    {this.state.user ? (
                        <Route path="/saved">
                            <Saved user={this.state.user} />
                        </Route>
                    ) : null}
                </Switch>
            </Layout>
        );
    }
}

export default withRouter(App);
