import React, { Component } from 'react';

import { NavLink } from 'react-router-dom';

import DrawerToggle from 'components/navigation/SideDrawer/DrawerToggle';
import SideDrawer from 'components/navigation/SideDrawer';
import Toolbar from 'components/navigation/Toolbar';

import styles from './index.module.css';

import FGLogo from 'assets/images/FGLogo.png';
import Logout from 'assets/images/Logout.png';

class Layout extends Component {
    state = {
        isSideDrawerOpen: false
    };

    sideDrawerHandler = () => {
        this.setState(prevState => {
           return {
               isSideDrawerOpen: !prevState.isSideDrawerOpen
           };
        });
    };

    render() {
        const links = [
            { to: '/', value: 'Home' },
            { to: '/generator', value: 'New font' }
        ];

        if (this.props.isAuth) {
            links.push({ to: '/saved', value: 'Saved fonts' });
        } else {
            links.push({ to: '/auth', value: 'Authenticate' });
        }

        return (
            <div className={styles.Layout}>
                <header>
                    <DrawerToggle click={this.sideDrawerHandler} />
                    <div className={styles.FGLogo}>
                        <NavLink to="/" exact><img src={FGLogo} alt="FG Logo"/></NavLink>
                    </div>
                    <SideDrawer isShown={this.state.isSideDrawerOpen} links={links} close={this.sideDrawerHandler} />
                    <Toolbar links={links} />
                    <div className={styles.Logout}>
                        <button onClick={this.props.logout}>
                            {this.props.logout ? <img src={Logout} alt="Logout" /> : null}
                        </button>
                    </div>
                </header>
                {this.props.children}
                <footer>
                    <p>&copy; 2019 - {new Date().getFullYear()} All rights reserved | Sami</p>
                </footer>
            </div>
        );
    }
}

export default Layout;