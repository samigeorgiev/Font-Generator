import React, { Component } from 'react';

import { Route, Switch } from 'react-router-dom';

import Auth from 'pages/Auth';
import Home from 'pages/Home';
import Layout from 'components/Layout';
import NewFont from 'pages/NewFont';

class App extends Component {
    state = {
        user: null
    };

    loginHandler = (token, expTime = 86400000) => {
        localStorage.setItem('token', token);
        localStorage.setItem('expDate', new Date(new Date().getTime() + +expTime));
        this.setState({
            user: {
                token: token
            }
        });
        setTimeout(() => {
            this.logoutHandler();
        }, expTime);
    };

    logoutHandler = () => {
        console.trace();
        localStorage.removeItem('token');
        localStorage.removeItem('expDate');
        this.setState({
            user: null
        });
    };

    componentDidMount() {
        const token = localStorage.getItem('token');
        const expDate = new Date(localStorage.getItem('expDate'));
        if (token && expDate > new Date().getTime()) {
            this.loginHandler(token, expDate - new Date().getTime());
        }
    }

    render() {
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
                        <NewFont />
                    </Route>
                    {!this.state.user
                        ? <Route path="/auth">
                            <Auth login={this.loginHandler} />
                        </Route>
                        : null}
                </Switch>
            </Layout>
        );
    }
}

export default App;