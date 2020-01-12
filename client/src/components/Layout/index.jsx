import React from 'react';

import { NavLink } from 'react-router-dom';

import Logo from 'components/Logo';
import Toolbar from 'components/navigation/Toolbar';

import styles from './index.module.css';

const layout = props => (
    <div className={styles.Layout}>
        <header>
            <div className={styles.HomeLink}>
                <NavLink to="/" exact><Logo /></NavLink>
            </div>
            <Toolbar />
            <Logo />
        </header>
        <main>
            {props.children}
        </main>
        <footer>
            <p>&copy; 2019 - {new Date().getFullYear()} All rights reserved | Sami</p>
        </footer>
    </div>
);

export default layout;